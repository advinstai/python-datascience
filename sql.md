## Instalando um servidor de banco de dados
```
pip install db-sqlite3
```

## Instalando uma interface via web para manipular o Banco de Dados
```
pip install sqlite-web
```

## Executando o servidor de banco de dados
```
sqlite3
```

## Executando o servidor de banco de dados Para controlar uma base de dados
```
sqlite3 survey.db
```

## Utilizando a interface Web para manipular um banco de dados
```
sqlite_web survey.db
```

## Exportando dados de uma tabela para um CSV
```
sqlite3 -header -csv /home/silvio/survey.db "select * from visited" > surveyDB.csv
```

Curso Self-paced de SQL: http://swcarpentry.github.io/sql-novice-survey/
